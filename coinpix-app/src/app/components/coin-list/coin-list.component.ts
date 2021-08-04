import { Component, OnInit } from '@angular/core';
import { RestApiService } from "../../shared/rest-api.service";

@Component({
  selector: 'app-coin-list',
  templateUrl: './coin-list.component.html',
  styleUrls: ['./coin-list.component.css']
})
export class CoinListComponent implements OnInit {

  Coin: any = [];

  public id: string = '0';

  ngOnInit(){
    this.id ='0';
    console.log(this.id);
    this.loadCoins();
  } 
  
  public onValChange(val: string) {
    this.id = val;
    console.log(this.id);
    this.loadCoins();
  }

  constructor(
    public restApi: RestApiService
  ) { }

  // Get coin list
  loadCoins() {
    return this.restApi.getCoins(this.id).subscribe((data: {}) => {
      this.Coin = data;
      console.log(this.Coin);
    })
  }

}
